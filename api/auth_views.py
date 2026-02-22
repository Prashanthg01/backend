# api/auth_views.py – Login, logout, current user for token auth

from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate


@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
def login(request):
    """
    POST body: { "username": "...", "password": "..." }
    Returns: { "token": "...", "user": { "username": "...", "id": ... } }
    """
    username = request.data.get('username')
    password = request.data.get('password')
    if not username or not password:
        return Response(
            {'error': 'username and password required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    user = authenticate(request, username=username, password=password)
    if user is None:
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    token, _ = Token.objects.get_or_create(user=user)
    return Response({
        'token': token.key,
        'user': {
            'id': user.pk,
            'username': user.username,
        }
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
def logout(request):
    """
    Delete the current user's token so it cannot be reused.
    """
    try:
        request.user.auth_token.delete()
    except Exception:
        pass
    return Response({'detail': 'Logged out.'}, status=status.HTTP_200_OK)


@api_view(['GET'])
def me(request):
    """
    Return current authenticated user info.
    """
    return Response({
        'user': {
            'id': request.user.pk,
            'username': request.user.username,
        }
    }, status=status.HTTP_200_OK)
